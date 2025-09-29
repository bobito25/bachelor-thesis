

(define (problem BW-rand-6)
(:domain blocksworld-4ops)
(:objects a b c d e f )
(:init
(handempty)
(on a f)
(ontable b)
(on c a)
(ontable d)
(on e c)
(ontable f)
(clear b)
(clear d)
(clear e)
)
(:goal
(and
(on a c)
(on b a)
(on e f))
)
)


