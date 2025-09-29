

(define (problem BW-rand-6)
(:domain blocksworld-4ops)
(:objects a b c d e f )
(:init
(handempty)
(on a d)
(on b e)
(ontable c)
(on d f)
(on e c)
(ontable f)
(clear a)
(clear b)
)
(:goal
(and
(on a d)
(on c a)
(on e c))
)
)


