

(define (problem BW-rand-6)
(:domain blocksworld-4ops)
(:objects a b c d e f )
(:init
(handempty)
(ontable a)
(on b a)
(ontable c)
(on d e)
(on e f)
(ontable f)
(clear b)
(clear c)
(clear d)
)
(:goal
(and
(on b e)
(on c a)
(on d b))
)
)


